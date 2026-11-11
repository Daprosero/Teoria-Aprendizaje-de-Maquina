/*
 * Matrix view of convolution (im2col): image windows become rows of X_patches,
 * filters become columns of W, and Y = X_patches · W reshapes into the feature maps.
 * Mount point: [data-matrix-lesson]. Requires spatial.js first (browser global
 * SpatialLesson, or require("./spatial.js") in Node); data, convolution and the
 * player/controls all come from there. Browser: global MatrixLesson (auto-mounts).
 */
(function (root, factory) {
  const spatial =
    root && root.SpatialLesson ? root.SpatialLesson : typeof require === "function" ? require("./spatial.js") : null;
  if (!spatial) throw new Error("MatrixLesson needs spatial.js loaded first");
  const api = factory(spatial);
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root && root.document) {
    root.MatrixLesson = api;
    const start = () => api.autoMount(root.document);
    if (root.document.readyState === "loading") {
      root.document.addEventListener("DOMContentLoaded", start);
    } else {
      start();
    }
  }
})(typeof window !== "undefined" ? window : null, function (S) {
  "use strict";

  const { HOUSE, FILTER_BANK, COLORS, windowAt, dotTerms, convolve2d } = S;
  const K = 3;
  const N = HOUSE.length - K + 1; // 3 window positions per axis, 9 windows
  const WINDOW_COLOR = "#B7791F";
  const FILTER_COLORS = [COLORS.cyan, COLORS.green, COLORS.rose];

  // ---------- Pure helpers ----------

  const shape = (m) => [m.length, m[0].length];

  // im2col layout (one window per row): windows in row-major order, each flattened row-major.
  function patchMatrix(image, k) {
    const n = image.length - k + 1;
    return Array.from({ length: n * n }, (_, p) => windowAt(image, Math.floor(p / n), p % n, k).flat());
  }

  // Each flattened kernel becomes one COLUMN of W.
  function filterMatrix(bank) {
    const cols = bank.map((filter) => filter.kernel.flat());
    return cols[0].map((_, i) => cols.map((col) => col[i]));
  }

  function matmul(A, B) {
    if (A[0].length !== B.length) throw new Error(`shape mismatch: ${shape(A)} · ${shape(B)}`);
    return A.map((row) => B[0].map((_, j) => row.reduce((s, x, i) => s + x * B[i][j], 0)));
  }

  function columnToMap(Y, f, n) {
    return Array.from({ length: n }, (_, r) => Array.from({ length: n }, (_, c) => Y[r * n + c][f]));
  }

  let cache = null;
  function matrixData() {
    if (!cache) {
      const X = patchMatrix(HOUSE, K);
      const W = filterMatrix(FILTER_BANK);
      // Direct maps are kept only so the figure and tests can compare both routes.
      cache = { X, W, Y: matmul(X, W), maps: FILTER_BANK.map((filter) => convolve2d(HOUSE, filter.kernel)) };
    }
    return cache;
  }

  // ---------- SVG primitives (local copies: spatial.js does not export its drawing helpers) ----------

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

  // Empty cells keep the matrix shape visible; only filled cells carry data-cell/data-value.
  function grid(o) {
    const filled = o.filled || (() => true);
    const font = Math.round(o.size * 0.55);
    let out = "";
    o.matrix.forEach((row, r) =>
      row.forEach((v, c) => {
        const x = o.x + c * o.size;
        const y = o.y + r * o.size;
        if (!filled(r, c)) {
          out += `<rect x="${x}" y="${y}" width="${o.size}" height="${o.size}" fill="${COLORS.empty}" stroke="${COLORS.paper}" stroke-width="1"/>`;
          return;
        }
        const tone = o.tone(v, c);
        const alpha = tone.alpha ? ` fill-opacity="${tone.alpha}"` : "";
        out +=
          `<rect x="${x}" y="${y}" width="${o.size}" height="${o.size}" fill="${tone.fill}"${alpha}` +
          ` stroke="${COLORS.paper}" stroke-width="1" data-cell="${o.prefix}-${r}-${c}" data-value="${v}"/>`;
        out += text(x + o.size / 2, y + o.size / 2 + font * 0.36, v, {
          size: font,
          anchor: "middle",
          fill: tone.dark ? COLORS.paper : COLORS.ink,
        });
      })
    );
    return out;
  }

  const pixelTone = (v) => ({ fill: v ? COLORS.ink : COLORS.paper, dark: v !== 0 });
  const weightTone = (v, c) => ({ fill: v ? FILTER_COLORS[c] : COLORS.paper, dark: v !== 0 });
  const outTone = (color) => (v) => ({ fill: color, alpha: (0.15 + 0.2 * v).toFixed(2) });
  const columnTone = (v, c) => outTone(FILTER_COLORS[c])(v);

  // ---------- Frame layout ----------

  const PHASES = { windows: 0, filters: N * N, products: N * N + FILTER_BANK.length };
  const FRAME_COUNT = PHASES.products + N * N * FILTER_BANK.length + 1;

  const L = { img: [16, 92, 22], small: [38, 226, 22], X: [146, 92, 20], W: [350, 92, 20], Y: [434, 92, 20] };
  const MAP = { x: 516, y: 96, size: 18, gap: 66 };

  function stageOf(frame) {
    if (frame < PHASES.filters) return { phase: "windows", p: frame, f: -1 };
    if (frame < PHASES.products) return { phase: "filters", p: -1, f: frame - PHASES.filters };
    if (frame < FRAME_COUNT - 1) {
      const i = frame - PHASES.products;
      return { phase: "products", p: i % (N * N), f: Math.floor(i / (N * N)) };
    }
    return { phase: "final", p: -1, f: -1 };
  }

  const rowText = (v) => `[${[0, 3, 6].map((i) => v.slice(i, i + 3).join(",")).join("  ")}]`;

  function renderMatrix(frame) {
    const { X, W, Y } = matrixData();
    const s = stageOf(frame);
    const { phase, p, f } = s;
    const r = Math.floor(p / N);
    const c = p % N;
    // How far each matrix has been built at this stage.
    const xRows = phase === "windows" ? p + 1 : 9;
    const wCols = phase === "windows" ? 0 : phase === "filters" ? f + 1 : 3;
    const yDone = phase === "products" ? f * 9 + p + 1 : phase === "final" ? 27 : 0;
    const yFilled = (row, col) => col * 9 + row < yDone;
    let b = "";

    const titles = {
      windows: "1 · Cada ventana 3×3 se aplana en una fila de X",
      filters: "2 · Cada filtro 3×3 se aplana en una columna de W",
      products: "3 · Y = X · W: fila de X por columna de W",
      final: "4 · Cada columna de Y se reacomoda en un mapa 3×3",
    };
    b += text(300, 28, titles[phase], { size: 16, weight: "bold", anchor: "middle" });
    b += text(71, 72, "Imagen 5×5", { size: 12.5, weight: "bold", anchor: "middle" });
    b += text(236, 72, "X_patches (9×9)", { size: 12.5, weight: "bold", anchor: "middle" });
    b += text(380, 72, "W (9×3)", { size: 12.5, weight: "bold", anchor: "middle" });
    b += text(464, 72, "Y (9×3)", { size: 12.5, weight: "bold", anchor: "middle" });
    b += text(543, 72, "3 mapas 3×3", { size: 12.5, weight: "bold", anchor: "middle" });
    [L.W, L.Y].forEach(([x]) =>
      FILTER_COLORS.forEach((color, i) => {
        b += text(x + i * 20 + 10, 88, `f${i + 1}`, { size: 10.5, anchor: "middle", fill: color, weight: "bold" });
      })
    );

    // Image, current window and the small patch or kernel underneath.
    const [ix, iy, is] = L.img;
    b += grid({ matrix: HOUSE, x: ix, y: iy, size: is, prefix: "img", tone: pixelTone });
    if (p >= 0) b += box(ix + c * is, iy + r * is, K * is, K * is, WINDOW_COLOR);
    const [sx, sy, ss] = L.small;
    if (phase === "windows") {
      b += text(71, sy - 6, `Ventana ${p + 1}`, { size: 12, anchor: "middle", fill: WINDOW_COLOR, weight: "bold" });
      b += grid({ matrix: windowAt(HOUSE, r, c, K), x: sx, y: sy, size: ss, prefix: "patch", tone: pixelTone });
      b += box(sx, sy, K * ss, K * ss, WINDOW_COLOR);
    } else if (f >= 0) {
      b += text(71, sy - 6, `Filtro ${f + 1}`, { size: 12, anchor: "middle", fill: FILTER_COLORS[f], weight: "bold" });
      b += grid({
        matrix: FILTER_BANK[f].kernel,
        x: sx,
        y: sy,
        size: ss,
        prefix: "kernel",
        tone: (v) => weightTone(v, f),
      });
    }

    // X_patches · W = Y, built stage by stage.
    const [xx, xy, xs] = L.X;
    b += grid({ matrix: X, x: xx, y: xy, size: xs, prefix: "x", filled: (row) => row < xRows, tone: pixelTone });
    if (p >= 0) b += box(xx, xy + p * xs, 9 * xs, xs, WINDOW_COLOR);
    b += text(338, 188, "×", { size: 20, anchor: "middle", fill: COLORS.muted });
    const [wx, wy, ws] = L.W;
    b += grid({ matrix: W, x: wx, y: wy, size: ws, prefix: "w", filled: (_, col) => col < wCols, tone: weightTone });
    if (f >= 0) b += box(wx + f * ws, wy, ws, 9 * ws, FILTER_COLORS[f]);
    b += text(422, 188, "=", { size: 20, anchor: "middle", fill: COLORS.muted });
    const [yx, yy, ys] = L.Y;
    b += grid({ matrix: Y, x: yx, y: yy, size: ys, prefix: "y", filled: yFilled, tone: columnTone });
    if (phase === "products") b += box(yx + f * ys, yy + p * ys, ys, ys, COLORS.ink);

    // Each column of Y, reshaped row-major, is one 3×3 map.
    FILTER_BANK.forEach((_, i) => {
      const my = MAP.y + i * MAP.gap;
      b += text(MAP.x, my - 4, `mapa ${i + 1}`, { size: 10.5, fill: FILTER_COLORS[i], weight: "bold" });
      b += grid({
        matrix: columnToMap(Y, i, N),
        x: MAP.x,
        y: my,
        size: MAP.size,
        prefix: `m${i}`,
        filled: (mr, mc) => yFilled(mr * N + mc, i),
        tone: outTone(FILTER_COLORS[i]),
      });
      if (phase === "products" && i === f) b += box(MAP.x + c * MAP.size, my + r * MAP.size, MAP.size, MAP.size, COLORS.ink);
    });

    const note = (lines, color) =>
      lines.forEach((str, i) => {
        b += text(300, 404 + i * 20, str, { size: 12, anchor: "middle", fill: color || COLORS.muted });
      });
    let label;
    if (phase === "windows") {
      b += text(300, 316, `ventana (${r + 1}, ${c + 1}) → fila ${p + 1} de X = ${rowText(X[p])}`, {
        size: 14,
        anchor: "middle",
        weight: "bold",
      });
      note(["Las ventanas se solapan: un mismo píxel aparece en varias filas de X.", "Paso 1, sin relleno: 3×3 posiciones = 9 filas de 9 valores."]);
      label = `Ventana ${p + 1}/9 (fila ${r + 1}, columna ${c + 1} de la imagen) → fila ${p + 1} de X: ${rowText(X[p])}.`;
    } else if (phase === "filters") {
      const col = W.map((row) => row[f]);
      b += text(300, 316, `filtro ${f + 1} (${FILTER_BANK[f].name}) → columna ${f + 1} de W = ${rowText(col)}ᵀ`, {
        size: 14,
        anchor: "middle",
        weight: "bold",
      });
      note(["W tiene una columna por filtro: 9 pesos × 3 filtros.", "El filtro 3 es una copia del 1 (ilustrativo, no entrenado)."]);
      label = `Filtro ${f + 1} (${FILTER_BANK[f].name}) aplanado → columna ${f + 1} de W: ${rowText(col)}.`;
    } else if (phase === "products") {
      const { terms, sum } = dotTerms([X[p]], [W.map((row) => row[f])]);
      const spans = terms
        .map((t) => {
          const color = t.w ? FILTER_COLORS[f] : COLORS.muted;
          return `<tspan fill="${color}" font-weight="${t.w ? "bold" : "normal"}">${t.x}·${t.w}</tspan>`;
        })
        .join(" + ");
      b +=
        `<text x="300" y="306" font-size="14" text-anchor="middle" fill="${COLORS.ink}">` +
        `${spans} <tspan fill="${COLORS.ink}" font-weight="bold">= ${sum}</tspan></text>`;
      b += text(300, 334, `Y[${p + 1}, ${f + 1}] = ${sum}  →  mapa ${f + 1} [${r + 1}, ${c + 1}]`, {
        size: 15,
        anchor: "middle",
        weight: "bold",
      });
      note(["Es el mismo XW de la capa densa, pero cada fila de X", "es una ventana de la imagen, no la imagen entera."]);
      label =
        `Producto: fila ${p + 1} de X · columna ${f + 1} de W = ${sum} → Y[${p + 1}, ${f + 1}] → ` +
        `mapa ${f + 1}[${r + 1}, ${c + 1}].`;
    } else {
      b += box(yx - 3, yy - 3, ys + 6, 9 * ys + 6, COLORS.rose, true);
      b += box(yx + 2 * ys - 3, yy - 3, ys + 6, 9 * ys + 6, COLORS.rose, true);
      b += box(MAP.x - 3, MAP.y - 3, 3 * MAP.size + 6, 3 * MAP.size + 6, COLORS.rose, true);
      b += box(MAP.x - 3, MAP.y + 2 * MAP.gap - 3, 3 * MAP.size + 6, 3 * MAP.size + 6, COLORS.rose, true);
      b += text(300, 306, "Y = X_patches · W: el mismo XW de la capa densa", { size: 15, anchor: "middle", weight: "bold" });
      b += text(300, 330, "Ahora cada fila de X es una ventana solapada de la imagen (im2col).", { size: 12.5, anchor: "middle" });
      b += text(300, 350, "Las bibliotecas pueden obtener el mismo resultado sin copiar las ventanas.", {
        size: 12.5,
        anchor: "middle",
        fill: COLORS.muted,
      });
      note(["Columnas 1 y 3 de Y iguales → mapa 3 = mapa 1: redundancia.", "El entrenamiento decide si los filtros salen distintos o repetidos."], COLORS.rose);
      label =
        "Cada columna de Y se reacomoda en un mapa 3×3. Y = XW, como en la capa densa, con ventanas " +
        "como filas. Columnas 1 y 3 iguales: redundancia.";
    }

    return {
      svg:
        `<svg class="spatial-svg matrix-svg" viewBox="0 0 600 440" xmlns="http://www.w3.org/2000/svg"` +
        ` role="img" aria-label="Convolución como producto matricial: ventanas por filtros"` +
        ` preserveAspectRatio="xMidYMid meet"><rect width="600" height="440" rx="12" fill="${COLORS.bg}"/>${b}</svg>`,
      label,
    };
  }

  const LESSON = { frameCount: FRAME_COUNT, intervalMs: 1100, title: "Convolución como XW", render: renderMatrix };
  // Registering in the shared registry reuses SpatialLesson's player, controls and keyboard handling.
  S.LESSONS.matrix = LESSON;

  function mount(el) {
    const instance = S.mount(el, "matrix");
    instance.root.classList.add("matrix-lesson");
    return instance;
  }

  function autoMount(doc) {
    return Array.from(doc.querySelectorAll("[data-matrix-lesson]")).map((el) => mount(el));
  }

  return {
    HOUSE,
    FILTER_BANK,
    PHASES,
    LESSON,
    shape,
    patchMatrix,
    filterMatrix,
    matmul,
    columnToMap,
    matrixData,
    mount,
    autoMount,
  };
});
