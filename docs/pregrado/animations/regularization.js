/*
 * CNN regularization and summary lessons: inverted dropout and the end-to-end pipeline.
 * Mount point: [data-regularization-lesson="dropout" | "pipeline"].
 * Requires spatial.js and operations.js first (browser globals SpatialLesson/OperationLesson,
 * or require() in Node): the house, the kernels, convolve/pool2d and the player come from there.
 * Browser: global RegularizationLesson (auto-mounts). Node: CommonJS exports for tests.
 */
(function (root, factory) {
  const load = (name, file) =>
    root && root[name] ? root[name] : typeof require === "function" ? require(file) : null;
  const spatial = load("SpatialLesson", "./spatial.js");
  const ops = load("OperationLesson", "./operations.js");
  if (!spatial || !ops) throw new Error("RegularizationLesson needs spatial.js and operations.js loaded first");
  const api = factory(spatial, ops);
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root && root.document) {
    root.RegularizationLesson = api;
    const start = () => api.autoMount(root.document);
    if (root.document.readyState === "loading") {
      root.document.addEventListener("DOMContentLoaded", start);
    } else {
      start();
    }
  }
})(typeof window !== "undefined" ? window : null, function (S, O) {
  "use strict";

  const { HOUSE, KERNELS, COLORS } = S;

  // ---------- Pure inverted dropout ----------

  // Recursively checks a vector/grid of finite numbers and returns its shape-preserving map.
  function mapShape(values, fn, name, path) {
    if (!Array.isArray(values) || values.length === 0) throw new TypeError(`${name} must be a non-empty array of finite numbers`);
    return values.map((v, i) => {
      const at = path.concat(i);
      if (Array.isArray(v)) return mapShape(v, fn, name, at);
      if (!Number.isFinite(v)) throw new TypeError(`${name} must be a non-empty array of finite numbers`);
      return fn(v, at);
    });
  }

  const pick = (m, at) => at.reduce((node, i) => (Array.isArray(node) ? node[i] : undefined), m);

  /*
   * Inverted dropout. Training: kept values (mask 1) are divided by (1 − rate), dropped values
   * (mask 0) become 0. The mask is explicit, so the lessons stay deterministic; real training
   * draws a new random mask on every pass. Inference: an identical copy, mask ignored, no rescale.
   * rate 0 drops nothing. Inputs are never mutated; the shape never changes.
   */
  function dropout(activations, options) {
    const { rate = 0.5, mask, training = true } = options || {};
    if (typeof rate !== "number" || !(rate >= 0 && rate < 1)) throw new RangeError("rate must be a number in [0, 1)");
    if (typeof training !== "boolean") throw new TypeError("training must be a boolean");
    if (!training || rate === 0) return mapShape(activations, (v) => v, "activations", []);
    if (!Array.isArray(mask)) throw new TypeError("training dropout needs an explicit mask of 0/1 values");
    const scale = 1 / (1 - rate);
    const out = mapShape(
      activations,
      (v, at) => {
        const m = pick(mask, at);
        if (m !== 0 && m !== 1) throw new RangeError("mask must match the activations shape and hold only 0 or 1");
        return m === 1 ? v * scale : 0;
      },
      "activations",
      []
    );
    // A mask with extra entries is a shape mismatch too.
    const mismatch = (m, at) => {
      if (!Number.isFinite(pick(activations, at))) throw new RangeError("mask must match the activations shape and hold only 0 or 1");
      return m;
    };
    mapShape(mask, mismatch, "mask", []);
    return out;
  }

  // ---------- Lesson data ----------

  const ACTIVATIONS = [2, 1, 3, 2, 4, 1];
  const RATE = 0.5;
  // Fixed illustrative masks: each pass differs, the drop count varies (3, 2, 3, 2) and
  // every activation is dropped in one pass and comes back in another.
  const MASKS = [
    [1, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1],
    [1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 1, 0],
  ];

  // Second filter: the canonical horizontal row with a negative row below it, so ReLU has work.
  const CONTRAST = [
    [0, 0, 0],
    [1, 1, 1],
    [-1, -1, -1],
  ];
  // Illustrative Dense 8×2 weights (row i = flattened input i, column j = class j); bias 0.
  const DENSE = [
    [1, 0],
    [1, -1],
    [0, 1],
    [0, 0],
    [1, 0],
    [-1, 1],
    [0, 1],
    [0, -1],
  ];

  const memo = (fn) => {
    let value = null;
    return () => value || (value = fn());
  };

  const dropoutData = memo(() => ({
    activations: ACTIVATIONS,
    rate: RATE,
    masks: MASKS,
    passes: MASKS.map((mask) => ({ mask, output: dropout(ACTIVATIONS, { rate: RATE, mask, training: true }) })),
    inference: dropout(ACTIVATIONS, { rate: RATE, training: false }),
  }));

  const pipelineData = memo(() => {
    // 6×6 canvas (house + a zero row and column) so the 4×4 maps split exactly into 2×2 windows.
    const image = S.padInto(HOUSE, 6, 0, 0);
    const filters = [
      { name: "vertical", hint: "el de siempre", kernel: KERNELS.vertical },
      { name: "contraste horizontal", hint: "con pesos negativos", kernel: CONTRAST },
    ];
    const conv = filters.map((f) => O.convolve(image, f.kernel, { stride: 1, padding: 0 }));
    const relu = conv.map((m) => m.map((row) => row.map((v) => Math.max(0, v))));
    const pooled = relu.map((m) => O.pool2d(m, { size: 2, stride: 2, mode: "max" }));
    // Flatten channel by channel: map 1 row by row, then map 2.
    const flat = pooled.flatMap((m) => m.flat());
    const scores = [0, 1].map((j) => flat.reduce((s, x, i) => s + x * DENSE[i][j], 0));
    const n = conv[0].length;
    const shapes = ["6×6", `${n}×${n}×2`, `${n}×${n}×2`, `${pooled[0].length}×${pooled[0].length}×2`, String(flat.length), String(scores.length)];
    return { image, filters, conv, relu, pooled, flat, weights: DENSE, scores, shapes };
  });

  // ---------- SVG primitives (local copies, like operations.js) ----------

  const fmt = (v) => String(Math.round(v * 100) / 100).replace("-", "−");
  const esc = (s) =>
    String(s).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);

  function text(x, y, str, a) {
    const o = Object.assign({ size: 13, fill: COLORS.ink, anchor: "start", weight: "normal" }, a);
    return (
      `<text x="${x}" y="${y}" font-size="${o.size}" fill="${o.fill}" text-anchor="${o.anchor}"` +
      ` font-weight="${o.weight}">${esc(str)}</text>`
    );
  }

  function box(x, y, w, h, color, extra) {
    return (
      `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="none" stroke="${color}"` +
      ` stroke-width="2.5"${extra || ""}/>`
    );
  }

  const line = (x1, y1, x2, y2, color, width, opacity) =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width || 1.5}"` +
    ` stroke-opacity="${opacity == null ? 1 : opacity}"/>`;

  // Same contract as operations.js: only filled cells carry data-cell/data-value.
  function grid(o) {
    const filled = o.filled || (() => true);
    const font = o.font || Math.max(9, Math.round(o.size * 0.45));
    let out = "";
    o.matrix.forEach((row, r) =>
      row.forEach((v, c) => {
        const x = o.x + c * o.size;
        const y = o.y + r * o.size;
        if (!filled(r, c)) {
          out += `<rect x="${x}" y="${y}" width="${o.size}" height="${o.size}" fill="${COLORS.empty}" stroke="${COLORS.paper}" stroke-width="1.5"/>`;
          return;
        }
        const t = o.tone(v, r, c);
        const alpha = t.alpha ? ` fill-opacity="${t.alpha}"` : "";
        const extra = o.attrs ? o.attrs(r, c) : "";
        out +=
          `<rect x="${x}" y="${y}" width="${o.size}" height="${o.size}" fill="${t.fill}"${alpha}` +
          ` stroke="${COLORS.paper}" stroke-width="1.5" data-cell="${o.prefix}-${r}-${c}" data-value="${v}"${extra}/>`;
        if (o.labels !== false) {
          out += text(x + o.size / 2, y + o.size / 2 + font * 0.36, fmt(v), {
            size: font,
            anchor: "middle",
            fill: t.ink || (t.dark ? COLORS.paper : COLORS.ink),
            weight: t.bold ? "bold" : "normal",
          });
        }
      })
    );
    return out;
  }

  function svg(title, body) {
    return (
      `<svg class="spatial-svg regularization-svg" viewBox="0 0 600 440" xmlns="http://www.w3.org/2000/svg"` +
      ` role="img" aria-label="${esc(title)}" preserveAspectRatio="xMidYMid meet">` +
      `<rect width="600" height="440" rx="12" fill="${COLORS.bg}"/>${body}</svg>`
    );
  }

  const centered = (lines, y0, attrs) =>
    lines.map((s, i) => text(300, y0 + i * 19, s, Object.assign({ size: 12, anchor: "middle", fill: COLORS.muted }, attrs))).join("");

  const pixelTone = (v) => ({ fill: v ? COLORS.ink : COLORS.paper, dark: v !== 0 });
  const signTone = (top) => (v) =>
    v < 0 ? { fill: COLORS.rose, alpha: (0.2 + 0.5 * (-v / top)).toFixed(2) } : { fill: COLORS.cyan, alpha: (0.12 + 0.6 * (v / top)).toFixed(2) };
  const weightTone = (w) => (w > 0 ? { fill: COLORS.cyan, dark: true } : w < 0 ? { fill: COLORS.rose, dark: true } : { fill: COLORS.paper });

  // ---------- Lesson: dropout during training, nothing at inference ----------

  const DX = 150;
  const DC = 48;
  const ROWS = { act: 52, mask: 128, out: 204 };

  function dropoutPhase(frame) {
    if (frame === 0) return { kind: "before" };
    if (frame >= 9) return { kind: "inference" };
    return { kind: "train", pass: Math.floor((frame - 1) / 2), scaled: frame % 2 === 0 };
  }

  function renderDropout(frame) {
    const d = dropoutData();
    const ph = dropoutPhase(frame);
    const train = ph.kind === "train";
    const mask = train ? d.masks[ph.pass] : null;
    const shown = !train ? d.inference : ph.scaled ? d.passes[ph.pass].output : ACTIVATIONS.map((a, i) => a * mask[i]);
    const top = Math.max(...d.passes.flatMap((p) => p.output));
    const titles = {
      before: "Antes del dropout: todas las activaciones pasan",
      train: train ? `Entrenamiento · pasada ${ph.pass + 1}/4 · p = ${RATE}` : "",
      inference: "Inferencia: sin máscaras, pasan todas",
    };
    let b = text(300, 28, titles[ph.kind], { size: 16, weight: "bold", anchor: "middle" });

    const outName = !train ? "Pasa a la capa" : ph.scaled ? "a × m ÷ (1 − p)" : "a × m";
    [
      ["Activaciones a", ROWS.act],
      ["Máscara m", ROWS.mask],
      [outName, ROWS.out],
    ].forEach(([name, y]) => (b += text(DX - 10, y + 30, name, { size: 12.5, weight: "bold", anchor: "end" })));

    b += grid({ matrix: [ACTIVATIONS], x: DX, y: ROWS.act, size: DC, prefix: "act", font: 18, tone: signTone(top) });
    if (train) {
      b += grid({
        matrix: [mask],
        x: DX,
        y: ROWS.mask,
        size: DC,
        prefix: "mask",
        font: 18,
        tone: (m) => (m ? { fill: COLORS.green, dark: true, bold: true } : { fill: COLORS.empty, ink: COLORS.muted }),
      });
    } else {
      b += `<rect x="${DX}" y="${ROWS.mask}" width="${6 * DC}" height="${DC}" rx="4" fill="none" stroke="${COLORS.line}" stroke-dasharray="6 4"/>`;
      b += text(DX + 3 * DC, ROWS.mask + 29, ph.kind === "inference" ? "sin máscara: no se apaga nada" : "todavía sin dropout", { size: 13, anchor: "middle", fill: COLORS.muted });
    }

    const dropped = (i) => train && mask[i] === 0;
    b += grid({
      matrix: [shown],
      x: DX,
      y: ROWS.out,
      size: DC,
      prefix: "out",
      font: 18,
      tone: (v, r, i) => (dropped(i) ? { fill: COLORS.empty, ink: COLORS.muted } : Object.assign(signTone(top)(v), { bold: train && ph.scaled })),
      attrs: (r, i) => (dropped(i) ? ' data-dropped="1"' : ""),
    });
    ACTIVATIONS.forEach((a, i) => {
      const x = DX + i * DC;
      if (dropped(i)) {
        b += line(x + 8, ROWS.out + 8, x + DC - 8, ROWS.out + DC - 8, COLORS.rose, 2.5);
        b += line(x + DC - 8, ROWS.out + 8, x + 8, ROWS.out + DC - 8, COLORS.rose, 2.5);
      } else {
        // Only surviving values reach the next layer; each frame redraws its own connections.
        b += line(x + DC / 2, ROWS.out + DC, 294, 318, COLORS.cyan, 2, 0.7);
        if (train && ph.scaled) b += text(x + DC / 2, ROWS.out + DC - 4, `${a} ÷ 0.5`, { size: 9, anchor: "middle", fill: COLORS.muted });
      }
    });
    b += `<circle cx="294" cy="338" r="20" fill="${COLORS.paper}" stroke="${COLORS.ink}" stroke-width="2"/>`;
    b += text(322, 342, "capa siguiente", { size: 12, fill: COLORS.muted });

    // Mask history: one row per pass already seen, so the changing masks stay visible.
    b += text(530, 62, "Máscaras usadas", { size: 12, weight: "bold", anchor: "middle" });
    const seen = ph.kind === "before" ? 0 : train ? ph.pass + 1 : 4;
    d.masks.forEach((m, k) => {
      const y = 74 + k * 30;
      b += text(474, y + 14, String(k + 1), { size: 11, anchor: "end", fill: COLORS.muted });
      b += grid({
        matrix: [m],
        x: 478,
        y,
        size: 18,
        prefix: `hist${k}`,
        filled: () => k < seen,
        tone: (v) => (v ? { fill: COLORS.green, dark: true } : { fill: COLORS.empty, ink: COLORS.muted }),
      });
      if (train && k === ph.pass) b += box(476, y - 2, 112, 22, COLORS.ink);
    });

    const notes = {
      before: ["Seis activaciones positivas de una capa: todas llegan a la capa siguiente.", "Dropout regulariza: reduce la co-adaptación entre neuronas;", "no garantiza mejores resultados."],
      train: [
        `p = ${RATE}: cada activación se apaga con probabilidad ${RATE}; cuántas, varía por pasada.`,
        "Las que siguen se dividen por (1 − p), o sea ÷ 0.5: valen el doble.",
        "Ninguna neurona se elimina: la apagada vuelve en otra pasada.",
      ],
      inference: ["Inferencia: sin máscara y sin reescalar.", "El ÷ (1 − p) ya se hizo al entrenar (dropout invertido).", "Dropout regulariza: reduce la co-adaptación; no garantiza mejores resultados."],
    };
    b += centered(notes[ph.kind], 372, { size: 12, fill: COLORS.ink });
    b += centered(["Máscaras fijas ilustrativas (en el entrenamiento real se sortean al azar) · forma 6 → 6"], 430, { size: 11 });

    let label;
    if (ph.kind === "before") label = `Antes del dropout: las 6 activaciones (${ACTIVATIONS.join(", ")}) pasan a la capa siguiente.`;
    else if (ph.kind === "inference") label = `Inferencia: sin máscara y sin reescalar; cada activación pasa tal cual (2 → 2). Siguen siendo 6 valores: ${ACTIVATIONS.join(", ")}.`;
    else {
      const off = mask.filter((m) => m === 0).length;
      label = ph.scaled
        ? `Entrenamiento, pasada ${ph.pass + 1}/4: las que siguen se dividen por (1 − 0.5), es decir ÷ 0.5. La primera: 2 → ${shown[0]}${mask[0] ? "" : " (apagada)"}. Salida: ${shown.join(", ")}.`
        : `Entrenamiento, pasada ${ph.pass + 1}/4: la máscara ilustrativa apaga ${off} de 6 activaciones; las otras ${6 - off} pasan.`;
    }
    return { svg: svg("Dropout: distintas activaciones apagadas en cada pasada de entrenamiento, ninguna en inferencia", b), label };
  }

  // ---------- Lesson: image → convolution → ReLU → pooling → flatten → scores ----------

  const STAGES = ["image", "conv", "relu", "pool", "flatten", "dense", "final"];
  const COL = {
    image: { x: 14, w: 72, title: "Imagen" },
    conv: { x: 102, w: 88, title: "Conv" },
    relu: { x: 210, w: 88, title: "ReLU" },
    pool: { x: 318, w: 60, title: "Max pool" },
    flatten: { x: 404, w: 22, title: "Aplanar" },
    dense: { x: 488, w: 40, title: "Densa" },
  };
  const MAP_Y = [84, 190];
  const POOL_Y = [98, 204];
  const FLAT_Y = 94;
  const SCORE_Y = 140;

  function renderPipeline(frame) {
    const p = pipelineData();
    const st = STAGES[frame];
    const at = STAGES.indexOf(st);
    const shown = (name) => at >= STAGES.indexOf(name);
    let b = text(300, 26, "De la imagen a los puntajes: cómo cambian los mapas", { size: 16, weight: "bold", anchor: "middle" });

    Object.keys(COL).forEach((name, i) => {
      const c = COL[name];
      const cx = c.x + c.w / 2;
      const on = name === st || st === "final";
      b += text(cx, 54, c.title, { size: 12.5, weight: "bold", anchor: "middle", fill: on ? COLORS.ink : COLORS.muted });
      b += text(cx, 70, p.shapes[i], { size: 12, anchor: "middle", fill: on ? COLORS.rose : COLORS.muted, weight: on ? "bold" : "normal" });
      if (name === st) b += box(c.x - 6, 40, c.w + 12, 244, COLORS.rose, ` data-stage="${name}"`);
    });
    [94, 198, 306, 391, 455].forEach((x) => (b += text(x, 186, "→", { size: 16, anchor: "middle", fill: COLORS.muted })));

    b += grid({ matrix: p.image, x: COL.image.x, y: 145, size: 12, prefix: "img", labels: false, tone: pixelTone });
    [0, 1].forEach((i) => {
      b += grid({ matrix: p.conv[i], x: COL.conv.x, y: MAP_Y[i], size: 22, prefix: `conv${i}`, filled: () => shown("conv"), tone: signTone(3) });
      b += grid({
        matrix: p.relu[i],
        x: COL.relu.x,
        y: MAP_Y[i],
        size: 22,
        prefix: `relu${i}`,
        filled: () => shown("relu"),
        tone: (v, r, c) => (p.conv[i][r][c] < 0 && st === "relu" ? { fill: COLORS.empty, ink: COLORS.rose, bold: true } : signTone(3)(v)),
        attrs: (r, c) => (p.conv[i][r][c] < 0 ? ' data-clamped="1"' : ""),
      });
      b += grid({ matrix: p.pooled[i], x: COL.pool.x, y: POOL_Y[i], size: 30, prefix: `pool${i}`, filled: () => shown("pool"), tone: signTone(3) });
    });
    b += grid({ matrix: p.flat.map((v) => [v]), x: COL.flatten.x, y: FLAT_Y, size: 22, prefix: "flat", filled: () => shown("flatten"), tone: signTone(3) });
    b += grid({ matrix: p.scores.map((v) => [v]), x: COL.dense.x, y: SCORE_Y, size: 40, prefix: "score", font: 16, filled: () => shown("dense"), tone: (v) => ({ fill: COLORS.green, alpha: (0.2 + 0.08 * v).toFixed(2), bold: true }) });
    ["clase 1", "clase 2"].forEach((s, j) => (b += text(COL.dense.x + 46, SCORE_Y + 25 + j * 40, s, { size: 12, fill: COLORS.muted })));

    // Selected-value links for the active stage.
    const link = (x1, y1, x2, y2) => line(x1, y1, x2, y2, COLORS.rose, 2, 0.85);
    if (st === "conv") {
      b += box(COL.image.x, 145, 36, 36, COLORS.rose);
      [0, 1].forEach((i) => (b += box(COL.conv.x, MAP_Y[i], 22, 22, COLORS.ink) + link(COL.image.x + 36, 163, COL.conv.x, MAP_Y[i] + 11)));
    } else if (st === "relu") {
      b += box(COL.conv.x, MAP_Y[1], 22, 22, COLORS.rose) + box(COL.relu.x, MAP_Y[1], 22, 22, COLORS.rose);
      b += link(COL.conv.x + 22, MAP_Y[1] + 11, COL.relu.x, MAP_Y[1] + 11);
    } else if (st === "pool") {
      b += box(COL.relu.x, MAP_Y[1], 44, 44, COLORS.rose) + box(COL.pool.x, POOL_Y[1], 30, 30, COLORS.ink);
      b += link(COL.relu.x + 44, MAP_Y[1] + 22, COL.pool.x, POOL_Y[1] + 15);
    } else if (st === "flatten") {
      p.flat.forEach((_, k) => {
        const [i, r, c] = [Math.floor(k / 4), Math.floor((k % 4) / 2), k % 2];
        b += line(COL.pool.x + c * 30 + 15, POOL_Y[i] + r * 30 + 15, COL.flatten.x, FLAT_Y + k * 22 + 11, COLORS.rose, 1.2, 0.6);
      });
    } else if (st === "dense") {
      p.weights.forEach((row, k) =>
        row.forEach((w, j) => {
          if (w) b += line(COL.flatten.x + 22, FLAT_Y + k * 22 + 11, COL.dense.x, SCORE_Y + j * 40 + 20, w > 0 ? COLORS.cyan : COLORS.rose, 1.5, 0.8);
        })
      );
    } else if (st === "final") {
      b += `<rect x="432" y="150" width="46" height="64" rx="4" fill="none" stroke="${COLORS.muted}" stroke-dasharray="5 4"/>`;
      b += text(455, 230, "dropout", { size: 10, anchor: "middle", fill: COLORS.muted });
    }

    const kernelPanel = () =>
      p.filters
        .map((f, i) => {
          const x = 70 + i * 250;
          return (
            grid({ matrix: f.kernel, x, y: 300, size: 22, prefix: `k${i}`, tone: weightTone }) +
            text(x + 76, 316, `Filtro ${i + 1}: ${f.name}`, { size: 12.5, weight: "bold" }) +
            text(x + 76, 334, f.hint, { size: 12, fill: COLORS.muted })
          );
        })
        .join("");
    const terms = (j) =>
      p.flat
        .map((x, i) => [x, p.weights[i][j]])
        .filter(([, w]) => w)
        .map(([x, w]) => (w < 0 ? `${x}·(${fmt(w)})` : `${x}·${w}`))
        .join(" + ");
    const notes = {
      image: ["Lienzo 6×6: la casa 5×5 + una fila y una columna de ceros, para que los mapas 4×4", "se agrupen justo de a 2×2. Los dos filtros todavía no recorrieron la imagen."],
      conv: ["Cada filtro recorre el lienzo con paso 1 y sin relleno: (6 − 3) + 1 = 4 → dos mapas 4×4.", "El filtro 2 da respuestas negativas (en rosa) donde hay píxeles debajo de la fila."],
      relu: ["ReLU = max(0, x), valor por valor: los negativos pasan a 0 (−2 → 0, −1 → 0).", "Los positivos no cambian y la forma tampoco: sigue siendo 4×4×2."],
      pool: ["Max pooling 2×2, paso 2: cada ventana deja su máximo. Ej.: max(0, 0, 2, 1) = 2.", "Cada mapa pasa de 4×4 a 2×2; los dos canales siguen separados: 2×2×2."],
      flatten: [`Aplanar: mapa 1 por filas y después mapa 2 → 8 valores: ${p.flat.join(", ")}.`, "El orden es una convención: lo importante es usar siempre el mismo."],
      dense: [
        `clase 1 = ${terms(0)} = ${p.scores[0]}`,
        `clase 2 = ${terms(1)} = ${p.scores[1]}`,
        "Puntajes crudos, sin softmax; los pesos 0 no aportan.",
        "No es una predicción de un modelo entrenado.",
      ],
      final: [p.shapes.join(" → "), "Dropout iría entre aplanar y densa, solo en entrenamiento;", "esto es inferencia: no se aplica."],
    };
    if (st === "image" || st === "conv") {
      b += kernelPanel();
      b += centered(notes[st], 372, { size: 12, fill: COLORS.ink });
    } else {
      b += centered(notes[st], 320, { size: st === "final" ? 13 : 12.5, fill: COLORS.ink, weight: st === "final" ? "bold" : "normal" });
      if (st === "final") b += centered(["Cada paso usa los valores del anterior; los números salen de las mismas funciones."], 386);
    }
    b += centered(["parámetros ilustrativos, no entrenados · filtros y pesos fijados a mano · sesgos 0"], 428, { size: 11 });

    const labels = {
      image: "Imagen: la casa en un lienzo 6×6. Dos filtros ilustrativos van a recorrerla.",
      conv: `Convolución: dos filtros 3×3, paso 1, sin relleno → dos mapas 4×4 (4×4×2). El filtro 2 tiene respuestas negativas.`,
      relu: "ReLU: cada negativo pasa a 0 (−2 → 0); los positivos quedan igual. Forma 4×4×2.",
      pool: "Max pooling 2×2, paso 2: cada ventana deja su max, por ejemplo max(0, 0, 2, 1) = 2. Forma 2×2×2.",
      flatten: `Aplanar: mapa 1 por filas, después mapa 2 → ${p.flat.join(", ")} (8 valores).`,
      dense: `Densa 8×2 con pesos ilustrativos: clase 1 = ${p.scores[0]}, clase 2 = ${p.scores[1]}. Son puntajes, no una predicción entrenada.`,
      final: `Recorrido completo: ${p.shapes.join(" → ")}. Dropout solo se usaría en entrenamiento.`,
    };
    return { svg: svg("Recorrido de una CNN: imagen, convolución, ReLU, pooling, aplanado y puntajes", b), label: labels[st] };
  }

  const LESSONS = {
    dropout: { frameCount: 10, intervalMs: 1800, title: "Dropout: no depender siempre de las mismas conexiones", render: renderDropout },
    pipeline: { frameCount: STAGES.length, intervalMs: 2600, title: "Recorrido completo de una CNN", render: renderPipeline },
  };
  // Registering in the shared registry reuses SpatialLesson's player, controls and keyboard handling.
  Object.keys(LESSONS).forEach((kind) => (S.LESSONS[`cnn-${kind}`] = LESSONS[kind]));

  function mount(el, kind) {
    const key = kind || el.getAttribute("data-regularization-lesson");
    if (!Object.prototype.hasOwnProperty.call(LESSONS, key)) throw new Error(`Unknown regularization lesson: ${key}`);
    const instance = S.mount(el, `cnn-${key}`);
    instance.root.classList.add("regularization-lesson");
    return instance;
  }

  function autoMount(doc) {
    return Array.from(doc.querySelectorAll("[data-regularization-lesson]"))
      .filter((el) => Object.prototype.hasOwnProperty.call(LESSONS, el.getAttribute("data-regularization-lesson")))
      .map((el) => mount(el));
  }

  return {
    ACTIVATIONS,
    MASKS,
    CONTRAST,
    DENSE,
    LESSONS,
    dropout,
    dropoutData,
    pipelineData,
    mount,
    autoMount,
  };
});
