/*
 * CNN geometry lessons: stride, zero padding, max pooling and average pooling.
 * Mount point: [data-operation-lesson="stride" | "padding" | "maxpool" | "averagepool"].
 * Requires spatial.js first (browser global SpatialLesson, or require("./spatial.js") in Node):
 * the house, the kernel, windowAt/dotTerms and the player/controls all come from there.
 * Browser: global OperationLesson (auto-mounts). Node: CommonJS exports for tests.
 */
(function (root, factory) {
  const spatial =
    root && root.SpatialLesson ? root.SpatialLesson : typeof require === "function" ? require("./spatial.js") : null;
  if (!spatial) throw new Error("OperationLesson needs spatial.js loaded first");
  const api = factory(spatial);
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root && root.document) {
    root.OperationLesson = api;
    const start = () => api.autoMount(root.document);
    if (root.document.readyState === "loading") {
      root.document.addEventListener("DOMContentLoaded", start);
    } else {
      start();
    }
  }
})(typeof window !== "undefined" ? window : null, function (S) {
  "use strict";

  const { HOUSE, KERNELS, COLORS, windowAt, dotTerms } = S;
  const KERNEL = KERNELS.vertical;
  const PAD_FILL = "#F2DDB0";
  const PAD_INK = "#8A5A12";

  // Illustrative non-negative activation map (e.g. after ReLU): one strong spike top-left,
  // a flat region bottom-right. Max and average pooling both read exactly this input.
  const POOL_INPUT = [
    [1, 3, 0, 2],
    [2, 9, 1, 1],
    [0, 1, 4, 4],
    [2, 3, 4, 4],
  ];

  // ---------- Pure geometry (square inputs only; that is all the lessons need) ----------

  function assertSquare(m, name) {
    const ok =
      Array.isArray(m) &&
      m.length > 0 &&
      m.every((row) => Array.isArray(row) && row.length === m.length && row.every(Number.isFinite));
    if (!ok) throw new TypeError(`${name} must be a non-empty square matrix of finite numbers`);
  }

  function assertInt(v, name, min) {
    if (!Number.isInteger(v) || v < min) {
      throw new RangeError(`${name} must be ${min > 0 ? "a positive" : "a non-negative"} integer`);
    }
  }

  // Output side for an n×n input, k×k window: floor((n + 2p − k) / s) + 1.
  function outputSize(n, k, stride, padding) {
    const p = padding || 0;
    assertInt(stride, "stride", 1);
    assertInt(p, "padding", 0);
    const span = n + 2 * p - k;
    if (span < 0) throw new RangeError(`window ${k}×${k} is larger than the padded ${n + 2 * p}×${n + 2 * p} input`);
    return Math.floor(span / stride) + 1;
  }

  // Top-left corners (in padded coordinates) of every window, row-major.
  function windowPositions(n, k, stride, padding) {
    const m = outputSize(n, k, stride, padding);
    return Array.from({ length: m * m }, (_, i) => ({
      row: Math.floor(i / m) * stride,
      col: (i % m) * stride,
    }));
  }

  function padZeros(image, padding) {
    assertSquare(image, "image");
    assertInt(padding, "padding", 0);
    const size = image.length + 2 * padding;
    return Array.from({ length: size }, (_, r) =>
      Array.from({ length: size }, (_, c) => {
        const row = image[r - padding];
        return row && c >= padding && c < padding + image.length ? row[c - padding] : 0;
      })
    );
  }

  // Deep-learning convention: cross-correlation, the kernel is not flipped; bias 0.
  function convolve(image, kernel, options) {
    const { stride = 1, padding = 0 } = options || {};
    assertSquare(image, "image");
    assertSquare(kernel, "kernel");
    const k = kernel.length;
    const m = outputSize(image.length, k, stride, padding);
    const padded = padZeros(image, padding);
    return Array.from({ length: m }, (_, r) =>
      Array.from({ length: m }, (_, c) => dotTerms(windowAt(padded, r * stride, c * stride, k), kernel).sum)
    );
  }

  // One pooling window: its values, which ones produce the result, and the result.
  function poolWindow(patch, mode) {
    if (mode !== "max" && mode !== "average") throw new RangeError("mode must be 'max' or 'average'");
    const values = patch.flat();
    const sum = values.reduce((s, v) => s + v, 0);
    const max = Math.max(...values);
    const chosen = values.map((v, i) => i).filter((i) => mode === "average" || values[i] === max);
    return { values, sum, value: mode === "max" ? max : sum / values.length, chosen };
  }

  // Valid pooling (no padding); no weights are involved.
  function pool2d(image, options) {
    const { size = 2, stride = 2, mode = "max" } = options || {};
    assertSquare(image, "image");
    assertInt(size, "size", 1);
    if (mode !== "max" && mode !== "average") throw new RangeError("mode must be 'max' or 'average'");
    const m = outputSize(image.length, size, stride, 0);
    return Array.from({ length: m }, (_, r) =>
      Array.from({ length: m }, (_, c) => poolWindow(windowAt(image, r * stride, c * stride, size), mode).value)
    );
  }

  const memo = (fn) => {
    let value = null;
    return () => value || (value = fn());
  };

  const strideData = memo(() => ({
    image: HOUSE,
    kernel: KERNEL,
    runs: [1, 2].map((stride) => ({
      stride,
      positions: windowPositions(HOUSE.length, KERNEL.length, stride),
      map: convolve(HOUSE, KERNEL, { stride }),
    })),
  }));

  const paddingData = memo(() => ({
    image: HOUSE,
    padded: padZeros(HOUSE, 1),
    valid: convolve(HOUSE, KERNEL),
    same: convolve(HOUSE, KERNEL, { padding: 1 }),
  }));

  const poolData = memo(() => ({
    input: POOL_INPUT,
    max: pool2d(POOL_INPUT, { mode: "max" }),
    average: pool2d(POOL_INPUT, { mode: "average" }),
  }));

  // ---------- SVG primitives (local copies, like matrix.js) ----------

  const fmt = (v) => String(Math.round(v * 100) / 100);
  const esc = (s) =>
    String(s).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);

  function text(x, y, str, a) {
    const o = Object.assign({ size: 13, fill: COLORS.ink, anchor: "start", weight: "normal" }, a);
    return (
      `<text x="${x}" y="${y}" font-size="${o.size}" fill="${o.fill}" text-anchor="${o.anchor}"` +
      ` font-weight="${o.weight}">${esc(str)}</text>`
    );
  }

  function box(x, y, w, h, color, dashed) {
    return (
      `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="none" stroke="${color}"` +
      ` stroke-width="3"${dashed ? ' stroke-dasharray="6 4"' : ""}/>`
    );
  }

  // The highlighted window carries its grid position so tests can read where it is.
  function windowBox(name, row, col, x, y, w, color) {
    return (
      `<rect data-window="${name}" data-row="${row}" data-col="${col}" x="${x}" y="${y}" width="${w}"` +
      ` height="${w}" rx="3" fill="none" stroke="${color}" stroke-width="3.5"/>`
    );
  }

  const arrow = (x1, y1, x2, y2, color) =>
    `<path d="M ${x1} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}" fill="none"` +
    ` stroke="${color}" stroke-width="2" stroke-dasharray="5 4"/>`;

  // Empty cells keep the shape visible; only filled cells carry data-cell/data-value.
  function grid(o) {
    const filled = o.filled || (() => true);
    const font = Math.max(9, Math.round(o.size * 0.45));
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

  const pixelTone = (v) => ({ fill: v ? COLORS.ink : COLORS.paper, dark: v !== 0 });
  const padTone = { fill: PAD_FILL, ink: PAD_INK };
  const mapTone = (color, top) => (v) => ({ fill: color, alpha: (0.15 + 0.6 * (v / top)).toFixed(2) });
  const byOrder = (n, count) => (r, c) => r * n + c < count;

  function svg(title, body) {
    return (
      `<svg class="spatial-svg operation-svg" viewBox="0 0 600 440" xmlns="http://www.w3.org/2000/svg"` +
      ` role="img" aria-label="${esc(title)}" preserveAspectRatio="xMidYMid meet">` +
      `<rect width="600" height="440" rx="12" fill="${COLORS.bg}"/>${body}</svg>`
    );
  }

  const centered = (lines, y0, attrs) =>
    lines.map((s, i) => text(300, y0 + i * 19, s, Object.assign({ size: 12, anchor: "middle", fill: COLORS.muted }, attrs))).join("");

  // ---------- Lesson: stride 1 vs stride 2 on the same house ----------

  const STRIDE_COLORS = [COLORS.cyan, COLORS.rose];

  function moveText(prev, cur, stride) {
    if (!prev) return "posición inicial";
    return prev.row === cur.row ? `→ avanza ${stride} píxel${stride > 1 ? "es" : ""}` : `↓ baja ${stride} fila${stride > 1 ? "s" : ""} y vuelve`;
  }

  function renderStride(frame) {
    const { runs } = strideData();
    const final = frame >= 9;
    let b = text(300, 26, "Mismo kernel, misma casa: solo cambia el paso (stride)", { size: 16, weight: "bold", anchor: "middle" });
    const parts = [];

    runs.forEach((run, i) => {
      const y0 = 44 + i * 182;
      const color = STRIDE_COLORS[i];
      const name = `s${run.stride}`;
      const n = run.map.length;
      const count = run.positions.length;
      const t = final ? count : Math.min(frame, count);
      const active = !final && frame < count;
      const pos = run.positions[t];
      b += text(20, y0 + 14, `Paso ${run.stride}`, { size: 15, weight: "bold", fill: color });
      b += grid({ matrix: HOUSE, x: 20, y: y0 + 24, size: 26, prefix: `${name}-img`, tone: pixelTone, labels: false });
      b += grid({ matrix: KERNEL, x: 196, y: y0 + 52, size: 16, prefix: `${name}-k`, tone: (w) => ({ fill: w ? color : COLORS.paper }), labels: false });
      b += text(220, y0 + 44, "kernel", { size: 11, anchor: "middle", fill: COLORS.muted });
      const outX = 324;
      const outY = y0 + 24 + (3 - n) * 17;
      b += grid({ matrix: run.map, x: outX, y: outY, size: 34, prefix: `${name}-out`, filled: byOrder(n, t + (active ? 1 : 0)), tone: mapTone(color, 3) });
      b += text(outX + (n * 34) / 2, y0 + 150, `mapa ${n}×${n}`, { size: 12, anchor: "middle", fill: COLORS.muted });

      if (active) {
        const prev = t > 0 ? run.positions[t - 1] : null;
        const value = run.map[Math.floor(t / n)][t % n];
        if (prev) b += box(20 + prev.col * 26, y0 + 24 + prev.row * 26, 78, 78, COLORS.muted, true);
        b += windowBox(name, pos.row, pos.col, 20 + pos.col * 26, y0 + 24 + pos.row * 26, 78, color);
        b += text(85, y0 + 170, moveText(prev, pos, run.stride), { size: 12, anchor: "middle", fill: color, weight: "bold" });
        b += arrow(20 + pos.col * 26 + 78, y0 + 24 + pos.row * 26 + 39, 196, y0 + 76, color);
        b += text(268, y0 + 82, `= ${value}`, { size: 18, weight: "bold", anchor: "middle", fill: color });
        b += arrow(286, y0 + 76, outX + (t % n) * 34, outY + Math.floor(t / n) * 34 + 17, color);
        b += box(outX + (t % n) * 34, outY + Math.floor(t / n) * 34, 34, 34, COLORS.ink);
        parts.push(`paso ${run.stride}: posición ${t + 1}/${count} → ${value}`);
      } else {
        b += text(85, y0 + 170, `terminó: ${count} posiciones`, { size: 12, anchor: "middle", fill: COLORS.muted });
        parts.push(`paso ${run.stride}: ya terminó sus ${count} posiciones`);
      }
      b += text(456, y0 + 60, `Salida ${n}×${n}`, { size: 15, weight: "bold", fill: color });
      b += text(456, y0 + 82, `(5 − 3) / ${run.stride} + 1 = ${n}`, { size: 13 });
      b += text(456, y0 + 102, `${count} posiciones`, { size: 12.5, fill: COLORS.muted });
    });

    b += text(300, 424, "Kernel vertical ilustrativo · sin relleno · tamaño = (n − k) / paso + 1", { size: 12, anchor: "middle", fill: COLORS.muted });
    const label = final
      ? "Paso 1 recorre 9 posiciones → mapa 3×3; paso 2 salta de a 2 píxeles, recorre 4 → mapa 2×2."
      : `Misma casa y kernel · ${parts.join(" · ")}.`;
    return { svg: svg("Paso 1 frente a paso 2 sobre la misma casa", b), label };
  }

  // ---------- Lesson: valid convolution shrinks, zero padding keeps 5×5 ----------

  function renderPadding(frame) {
    const { padded, valid, same } = paddingData();
    const phase = frame < 9 ? "valid" : frame < 34 ? "same" : "final";
    const p = phase === "valid" ? 0 : 1;
    const t = phase === "valid" ? frame : phase === "same" ? frame - 9 : 24;
    const input = p ? padded : HOUSE;
    const map = p ? same : valid;
    const n = map.length;
    const r = Math.floor(t / n);
    const c = t % n;
    const isPad = (row, col) => p > 0 && (row < p || col < p || row >= 5 + p || col >= 5 + p);
    const [X0, Y0, CS] = [16, 56, 28];
    const ix = X0 + (1 - p) * CS;
    let b = "";

    const titles = {
      valid: "1 · Sin relleno: la ventana no sale del borde, el mapa se achica",
      same: "2 · Relleno de 1 cero: la ventana se centra también en el borde",
      final: "3 · Con relleno 1 el mapa conserva 5×5",
    };
    b += text(300, 26, titles[phase], { size: 15, weight: "bold", anchor: "middle" });
    b += text(114, 48, p ? "Entrada con relleno 7×7" : "Imagen 5×5", { size: 12.5, weight: "bold", anchor: "middle" });
    b += grid({
      matrix: input,
      x: ix,
      y: Y0 + (1 - p) * CS,
      size: CS,
      prefix: "in",
      tone: (v, row, col) => (isPad(row, col) ? padTone : pixelTone(v)),
      attrs: (row, col) => (isPad(row, col) ? ' data-pad="1"' : ""),
    });
    if (p) b += box(X0 + CS, Y0 + CS, 5 * CS, 5 * CS, COLORS.muted, true);

    const patch = windowAt(input, r, c, 3);
    const { sum } = dotTerms(patch, KERNEL);
    const zeros = [0, 1, 2].flatMap((i) => [0, 1, 2].filter((j) => isPad(r + i, c + j))).length;
    if (phase !== "final") {
      b += windowBox("pad", r, c, ix + c * CS, Y0 + (1 - p) * CS + r * CS, 3 * CS, COLORS.cyan);
      b += text(266, 76, "Ventana", { size: 12, anchor: "middle", weight: "bold" });
      b += text(362, 76, "Kernel", { size: 12, anchor: "middle", weight: "bold" });
      b += grid({
        matrix: patch,
        x: 230,
        y: 86,
        size: 24,
        prefix: "patch",
        tone: (v, i, j) => (isPad(r + i, c + j) ? padTone : pixelTone(v)),
        attrs: (i, j) => (isPad(r + i, c + j) ? ' data-pad="1"' : ""),
      });
      b += text(314, 132, "×", { size: 20, anchor: "middle", fill: COLORS.muted });
      b += grid({ matrix: KERNEL, x: 326, y: 86, size: 24, prefix: "k", tone: (w) => ({ fill: w ? COLORS.cyan : COLORS.paper, dark: w !== 0 }) });
      b += text(314, 190, `suma = ${sum}`, { size: 16, anchor: "middle", weight: "bold", fill: COLORS.rose });
      if (zeros) b += text(314, 210, `${zeros} ceros de relleno`, { size: 12, anchor: "middle", fill: PAD_INK });
      b += arrow(370, 184, 432 + (1 - p) * 30 + c * 30, 76 + (1 - p) * 30 + r * 30 + 15, COLORS.rose);
    }
    b += text(507, 64, `Mapa ${n}×${n}`, { size: 12.5, weight: "bold", anchor: "middle" });
    const ox = 432 + (1 - p) * 30;
    const oy = 76 + (1 - p) * 30;
    b += grid({ matrix: map, x: ox, y: oy, size: 30, prefix: "out", filled: byOrder(n, t + 1), tone: mapTone(COLORS.green, 3) });
    if (phase !== "final") b += box(ox + c * 30, oy + r * 30, 30, 30, COLORS.rose);
    else b += box(462 - 3, 106 - 3, 96, 96, COLORS.rose, true);

    // Legend: padded zeros are not image pixels, even though they hold the value 0.
    [
      [PAD_FILL, "0 de relleno (agregado)"],
      [COLORS.paper, "píxel real = 0"],
      [COLORS.ink, "píxel real = 1"],
    ].forEach(([fill, str], i) => {
      const lx = 24 + i * 190;
      b += `<rect x="${lx}" y="268" width="16" height="16" fill="${fill}" stroke="${COLORS.line}"/>`;
      b += text(lx + 22, 281, str, { size: 12, fill: i === 0 ? PAD_INK : COLORS.ink });
    });
    b += text(300, 318, "Sin relleno: 5×5 → 3×3 · 9 salidas", { size: 14, anchor: "middle", weight: p ? "normal" : "bold", fill: p ? COLORS.muted : COLORS.ink });
    b += text(300, 342, "Con relleno 1: 7×7 → 5×5 · 25 salidas", { size: 14, anchor: "middle", weight: p ? "bold" : "normal", fill: p ? COLORS.green : COLORS.muted });
    const notes = {
      valid: ["Kernel 3×3, paso 1: (5 − 3) / 1 + 1 = 3. Cada capa así recorta el borde."],
      same: ["(5 + 2·1 − 3) / 1 + 1 = 5. Los ceros entran en la suma pero valen 0:", "solo permiten ubicar la ventana sobre los píxeles del borde."],
      final: ["El centro 3×3 del mapa (punteado) es exactamente el mapa sin relleno.", "Los ceros no agregan información real de la imagen: solo conservan el tamaño."],
    };
    b += centered(notes[phase], 384);

    let label;
    if (phase === "valid") label = `Sin relleno, posición ${t + 1}/9: ventana · kernel = ${sum} → mapa[${r + 1}, ${c + 1}]. La salida es 3×3.`;
    else if (phase === "same") label = `Con relleno 1, posición ${t + 1}/25: la ventana incluye ${zeros} ceros de relleno; suma = ${sum} → mapa[${r + 1}, ${c + 1}].`;
    else label = "Con relleno 1 el mapa queda 5×5 y su centro 3×3 coincide con el mapa sin relleno. Los ceros de relleno no agregan información de la imagen.";
    return { svg: svg("Relleno con ceros: el mapa conserva su tamaño", b), label };
  }

  // ---------- Lessons: max and average pooling over the same 4×4 map ----------

  function poolStage(frame) {
    const w = Math.min(Math.floor(frame / 2), 3);
    return { w, final: frame >= 8, pick: frame % 2 === 1, done: frame >= 8 ? 4 : Math.floor((frame + 1) / 2), r: Math.floor(w / 2) * 2, c: (w % 2) * 2 };
  }

  function poolInput(s, mode, x, y, size) {
    const result = poolWindow(windowAt(POOL_INPUT, s.r, s.c, 2), mode);
    const inWindow = (row, col) => !s.final && row >= s.r && row < s.r + 2 && col >= s.c && col < s.c + 2;
    const isChosen = (row, col) => inWindow(row, col) && s.pick && result.chosen.includes((row - s.r) * 2 + (col - s.c));
    let b = text(x + size * 2, y - 12, "Mapa de activación 4×4", { size: 12.5, weight: "bold", anchor: "middle" });
    b += grid({
      matrix: POOL_INPUT,
      x,
      y,
      size,
      prefix: "in",
      tone: (v, row, col) =>
        isChosen(row, col) && mode === "max"
          ? { fill: COLORS.rose, dark: true, bold: true }
          : { fill: COLORS.cyan, alpha: (inWindow(row, col) ? 0.35 + 0.6 * (v / 9) : 0.1 + 0.3 * (v / 9)).toFixed(2), bold: inWindow(row, col) },
      attrs: (row, col) => (isChosen(row, col) && mode === "max" ? ' data-chosen="1"' : ""),
    });
    if (!s.final) b += windowBox("pool", s.r, s.c, x + s.c * size, y + s.r * size, 2 * size, COLORS.rose);
    return { b, result };
  }

  const poolNote = "Mapa ilustrativo · ventana 2×2 · paso 2 · sin relleno · sin pesos que aprender";

  function renderMaxPool(frame) {
    const s = poolStage(frame);
    const { max } = poolData();
    const { b: inputSvg, result } = poolInput(s, "max", 40, 76, 52);
    const list = result.values.join(", ");
    let b = text(300, 28, s.final ? "Salida 2×2: la respuesta más fuerte de cada región" : "Max pooling: conservar la respuesta más fuerte", { size: 16, weight: "bold", anchor: "middle" });
    b += inputSvg;
    b += text(500, 64, "Salida 2×2", { size: 12.5, weight: "bold", anchor: "middle" });
    b += grid({ matrix: max, x: 440, y: 76, size: 60, prefix: "out", filled: byOrder(2, s.done), tone: mapTone(COLORS.rose, 9) });
    if (!s.final) {
      const [oy, ox] = [76 + (s.r / 2) * 60, 440 + (s.c / 2) * 60];
      b += text(344, 150, `max(${list})`, { size: 15, anchor: "middle" });
      b += text(344, 182, s.pick ? `= ${result.value}` : "= ?", { size: 20, anchor: "middle", weight: "bold", fill: COLORS.rose });
      if (s.pick && result.chosen.length > 1) b += text(344, 204, "empate: el valor es el mismo", { size: 11.5, anchor: "middle", fill: COLORS.muted });
      if (s.pick) b += arrow(372, 176, ox, oy + 30, COLORS.rose);
      b += box(ox, oy, 60, 60, COLORS.ink);
      b += centered(["¿Apareció una señal fuerte en esta región?", "Se queda solo el mayor de los cuatro valores."], 330, { size: 13 });
    } else {
      b += centered(["Cada número responde: ¿apareció una señal fuerte en esta región?", "Da cierta tolerancia a corrimientos pequeños dentro de la ventana, pero no la garantiza."], 330, { size: 12.5 });
    }
    b += centered([poolNote], 414);
    const cell = `salida[${s.r / 2 + 1}, ${s.c / 2 + 1}]`;
    const label = s.final
      ? "Salida 2×2 del max pooling: [[9, 2], [3, 4]]. Cada valor es el mayor de su ventana 2×2; no hay pesos que aprender."
      : s.pick
      ? `Ventana ${s.w + 1}/4: max(${list}) = ${result.value} → ${cell}.`
      : `Ventana ${s.w + 1}/4: sus cuatro valores son ${list}.`;
    return { svg: svg("Max pooling 2×2 con paso 2 sobre un mapa 4×4", b), label };
  }

  function renderAveragePool(frame) {
    const s = poolStage(frame);
    const { max, average } = poolData();
    const { b: inputSvg, result } = poolInput(s, "average", 24, 76, 46);
    const sumText = `(${result.values.join(" + ")}) / 4`;
    let b = text(300, 28, s.final ? "Misma entrada: promedio frente a máximo" : "Average pooling: resumir la respuesta de la región", { size: 16, weight: "bold", anchor: "middle" });
    b += inputSvg;
    b += text(356, 64, "Average pooling", { size: 12.5, weight: "bold", anchor: "middle", fill: COLORS.green });
    b += grid({ matrix: average, x: 300, y: 76, size: 56, prefix: "out", filled: byOrder(2, s.done), tone: mapTone(COLORS.green, 9) });
    b += text(516, 64, "Max (misma entrada)", { size: 12.5, weight: "bold", anchor: "middle", fill: COLORS.rose });
    b += grid({ matrix: max, x: 460, y: 76, size: 56, prefix: "max", tone: mapTone(COLORS.rose, 9) });
    if (!s.final) {
      const [dr, dc] = [(s.r / 2) * 56, (s.c / 2) * 56];
      b += box(300 + dc, 76 + dr, 56, 56, COLORS.ink);
      b += box(460 + dc, 76 + dr, 56, 56, COLORS.rose, true);
      const tail = s.pick ? ` = ${result.sum} / 4 = ${fmt(result.value)}` : " = ?";
      b += text(300, 290, sumText + tail, { size: 17, anchor: "middle", weight: "bold", fill: COLORS.green });
      if (s.pick) b += text(300, 316, `max pooling, en la misma ventana: ${max[s.r / 2][s.c / 2]}`, { size: 13, anchor: "middle", fill: COLORS.rose });
      b += centered(["Los cuatro valores cuentan igual: un pico se reparte entre ellos."], 350, { size: 12.5 });
    } else {
      b += box(300 - 3, 76 - 3, 62, 62, COLORS.rose, true);
      b += box(460 - 3, 76 - 3, 62, 62, COLORS.rose, true);
      b += centered(["Región con un pico (9): máximo 9, promedio 3.75; el promedio diluye el pico.", "Región pareja (4, 4, 4, 4): ambos dan 4. Ninguno de los dos aprende pesos."], 290, { size: 12.5, fill: COLORS.ink });
      b += centered(["Max: ¿hubo una señal fuerte? · Promedio: ¿cuánta señal hay en total?"], 350, { size: 12.5 });
    }
    b += centered([poolNote], 414);
    const cell = `salida[${s.r / 2 + 1}, ${s.c / 2 + 1}]`;
    const label = s.final
      ? "Misma entrada: max pooling conserva el pico 9; average pooling lo resume en 3.75. En la región pareja ambos dan 4."
      : s.pick
      ? `Ventana ${s.w + 1}/4: ${sumText} = ${result.sum} / 4 = ${fmt(result.value)} → ${cell}; max pooling daría ${max[s.r / 2][s.c / 2]}.`
      : `Ventana ${s.w + 1}/4: se suman sus cuatro valores (${result.values.join(", ")}) y se divide por 4.`;
    return { svg: svg("Average pooling 2×2 con paso 2 comparado con max pooling", b), label };
  }

  const LESSONS = {
    stride: { frameCount: 10, intervalMs: 1400, title: "Paso (stride): cuánto avanza la ventana", render: renderStride },
    padding: { frameCount: 35, intervalMs: 800, title: "Relleno (padding): qué pasa en los bordes", render: renderPadding },
    maxpool: { frameCount: 9, intervalMs: 1400, title: "Max pooling", render: renderMaxPool },
    averagepool: { frameCount: 9, intervalMs: 1400, title: "Average pooling", render: renderAveragePool },
  };
  // Registering in the shared registry reuses SpatialLesson's player, controls and keyboard handling.
  Object.keys(LESSONS).forEach((kind) => (S.LESSONS[`cnn-${kind}`] = LESSONS[kind]));

  function mount(el, kind) {
    const key = kind || el.getAttribute("data-operation-lesson");
    if (!Object.prototype.hasOwnProperty.call(LESSONS, key)) throw new Error(`Unknown operation lesson: ${key}`);
    const instance = S.mount(el, `cnn-${key}`);
    instance.root.classList.add("operation-lesson");
    return instance;
  }

  function autoMount(doc) {
    return Array.from(doc.querySelectorAll("[data-operation-lesson]"))
      .filter((el) => Object.prototype.hasOwnProperty.call(LESSONS, el.getAttribute("data-operation-lesson")))
      .map((el) => mount(el));
  }

  return {
    HOUSE,
    KERNEL,
    POOL_INPUT,
    LESSONS,
    outputSize,
    windowPositions,
    padZeros,
    convolve,
    poolWindow,
    pool2d,
    strideData,
    paddingData,
    poolData,
    mount,
    autoMount,
  };
});
