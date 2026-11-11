/*
 * Spatial convolution lessons: original, dependency-free animated SVG widgets.
 * Mount points: [data-spatial-lesson="comparison" | "kernel" | "filters"].
 * Browser: global SpatialLesson (auto-mounts). Node: CommonJS exports for tests.
 * All numbers drawn in the figures come from the same pure helpers the tests use.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root && root.document) {
    root.SpatialLesson = api;
    const start = () => api.autoMount(root.document);
    if (root.document.readyState === "loading") {
      root.document.addEventListener("DOMContentLoaded", start);
    } else {
      start();
    }
  }
})(typeof window !== "undefined" ? window : null, function () {
  "use strict";

  // ---------- Canonical data (feature doc L6; illustrative, not trained) ----------

  const HOUSE = [
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
  ];

  const KERNELS = {
    vertical: [
      [0, 1, 0],
      [0, 1, 0],
      [0, 1, 0],
    ],
    horizontal: [
      [0, 0, 0],
      [1, 1, 1],
      [0, 0, 0],
    ],
  };

  const copy = (m) => m.map((row) => row.slice());

  // The third filter is a separate copy so its map is genuinely computed again.
  const FILTER_BANK = [
    { name: "vertical", hint: "responde a paredes", kernel: copy(KERNELS.vertical) },
    { name: "horizontal", hint: "responde a techo y piso", kernel: copy(KERNELS.horizontal) },
    { name: "vertical", hint: "copia exacta del filtro 1", kernel: copy(KERNELS.vertical) },
  ];

  const COLORS = {
    bg: "#F4F7FA",
    ink: "#1B2029",
    cyan: "#347E98",
    rose: "#A83A59",
    green: "#66862E",
    muted: "#5B6472",
    line: "#C9D2DC",
    empty: "#E6EBF0",
    paper: "#FFFFFF",
  };

  // ---------- Pure helpers ----------

  function padInto(image, size, top, left) {
    const out = Array.from({ length: size }, () => new Array(size).fill(0));
    image.forEach((row, r) => row.forEach((v, c) => (out[r + top][c + left] = v)));
    return out;
  }

  function windowAt(image, row, col, k) {
    return image.slice(row, row + k).map((line) => line.slice(col, col + k));
  }

  function dotTerms(patch, kernel) {
    const terms = [];
    patch.forEach((row, r) =>
      row.forEach((x, c) => terms.push({ x, w: kernel[r][c], product: x * kernel[r][c] }))
    );
    return { terms, sum: terms.reduce((s, t) => s + t.product, 0) };
  }

  // Valid convolution (deep-learning convention: no kernel flip), stride 1.
  function convolve2d(image, kernel, bias) {
    const k = kernel.length;
    const n = image.length - k + 1;
    return Array.from({ length: n }, (_, r) =>
      Array.from({ length: n }, (_, c) => dotTerms(windowAt(image, r, c, k), kernel).sum + (bias || 0))
    );
  }

  function denseParams(inputs, outputs) {
    return { perNeuron: inputs + 1, neurons: outputs, total: (inputs + 1) * outputs };
  }

  function convParams(k, channels, filters) {
    const perFilter = k * k * channels + 1;
    return { perFilter, filters, total: perFilter * filters };
  }

  // Deterministic stand-ins for independently learned dense weights in [-1, 1].
  function denseWeights(neuron, inputs) {
    return Array.from({ length: inputs }, (_, i) => {
      const x = Math.sin((neuron + 1) * 12.9898 + (i + 1) * 78.233) * 43758.5453;
      return Math.round(((x - Math.floor(x)) * 2 - 1) * 10) / 10;
    });
  }

  let comparisonCache = null;
  function comparisonData() {
    if (!comparisonCache) {
      const canvases = [padInto(HOUSE, 7, 1, 0), padInto(HOUSE, 7, 1, 1)];
      comparisonCache = {
        canvases,
        kernel: KERNELS.vertical,
        maps: canvases.map((img) => convolve2d(img, KERNELS.vertical)),
        weights: Array.from({ length: 25 }, (_, j) => denseWeights(j, 49)),
      };
    }
    return comparisonCache;
  }

  // ---------- SVG primitives ----------

  const esc = (s) =>
    String(s).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);

  function text(x, y, str, attrs) {
    const a = Object.assign({ size: 13, fill: COLORS.ink, anchor: "start", weight: "normal" }, attrs);
    return (
      `<text x="${x}" y="${y}" font-size="${a.size}" fill="${a.fill}" text-anchor="${a.anchor}"` +
      ` font-weight="${a.weight}">${esc(str)}</text>`
    );
  }

  function frameRect(x, y, w, h, color, dashed) {
    return (
      `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="none" stroke="${color}"` +
      ` stroke-width="3"${dashed ? ' stroke-dasharray="6 4"' : ""}/>`
    );
  }

  function line(x1, y1, x2, y2, color, width, opacity) {
    return (
      `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}"` +
      ` stroke-width="${width || 1.5}" stroke-opacity="${opacity == null ? 1 : opacity}"/>`
    );
  }

  /*
   * Draws a matrix as square cells. Only "filled" cells carry data-cell/data-value,
   * which is what the tests read back from the rendered frame.
   */
  function cellGrid(o) {
    const filled = o.filled || (() => true);
    const fontSize = Math.max(9, Math.round(o.size * 0.45));
    let out = "";
    o.matrix.forEach((row, r) =>
      row.forEach((v, c) => {
        const x = o.x + c * o.size;
        const y = o.y + r * o.size;
        const on = filled(r, c);
        const data = on && o.prefix ? ` data-cell="${o.prefix}-${r}-${c}" data-value="${v}"` : "";
        const fill = on ? o.fill(v, r, c) : COLORS.empty;
        const alpha = on && o.alpha ? ` fill-opacity="${o.alpha(v, r, c)}"` : "";
        out +=
          `<rect x="${x}" y="${y}" width="${o.size}" height="${o.size}" fill="${fill}"${alpha}` +
          ` stroke="${COLORS.paper}" stroke-width="1.5"${data}/>`;
        if (on && o.labels) {
          const dark = o.textOn ? o.textOn(v) : false;
          out += text(x + o.size / 2, y + o.size / 2 + fontSize * 0.36, v, {
            size: fontSize,
            anchor: "middle",
            fill: dark ? COLORS.paper : COLORS.ink,
          });
        }
      })
    );
    return out;
  }

  const pixelFill = (v) => (v ? COLORS.ink : COLORS.paper);
  const weightFill = (color) => (w) => (w ? color : COLORS.paper);

  function svg(title, body) {
    return (
      `<svg class="spatial-svg" viewBox="0 0 600 440" xmlns="http://www.w3.org/2000/svg"` +
      ` role="img" aria-label="${esc(title)}" preserveAspectRatio="xMidYMid meet">` +
      `<rect width="600" height="440" rx="12" fill="${COLORS.bg}"/>${body}</svg>`
    );
  }

  // ---------- Lesson 1: dense vs Conv2D weight reuse ----------

  function renderComparison(frame) {
    const data = comparisonData();
    const phase = frame < 25 ? 0 : 1;
    const pos = frame % 25;
    const r = Math.floor(pos / 5);
    const c = pos % 5;
    const input = data.canvases[phase];
    const map = data.maps[phase];
    const weights = data.weights[pos];
    const dense = denseParams(49, 25);
    const conv = convParams(3, 1, 1);
    const maxMap = Math.max(1, ...map.flat());
    const shiftTag = phase ? "Casa desplazada 1 px →" : "Casa original";
    let b = line(300, 14, 300, 426, COLORS.line, 1.5);

    // Dense half: every output neuron sees all 49 pixels with its own weights.
    b += text(150, 28, "Densa", { size: 17, weight: "bold", anchor: "middle" });
    b += text(150, 45, "7×7 → 25 salidas · cada una ve los 49 píxeles", {
      size: 11.5,
      fill: COLORS.muted,
      anchor: "middle",
    });
    b += cellGrid({ matrix: input, x: 20, y: 58, size: 22, fill: pixelFill });
    b += frameRect(20, 58, 154, 154, COLORS.rose);
    const nx = 200 + c * 18 + 9;
    const ny = 90 + r * 18 + 9;
    for (let i = 0; i < 7; i += 1) b += line(174, 58 + i * 22 + 11, nx, ny, COLORS.rose, 1, 0.45);
    b += cellGrid({
      matrix: Array.from({ length: 5 }, (_, i) => Array.from({ length: 5 }, (_, j) => i * 5 + j)),
      x: 200,
      y: 90,
      size: 18,
      prefix: "dense-out",
      fill: (v) => (v === pos ? COLORS.rose : v < pos ? "#E9C9D3" : COLORS.paper),
    });
    b += text(245, 124 + 80, "25 neuronas", { size: 11.5, fill: COLORS.muted, anchor: "middle" });
    b += text(20, 230, shiftTag, { size: 11.5, fill: phase ? COLORS.rose : COLORS.muted });
    b += text(20, 252, `Pesos de la neurona ${pos + 1}`, { size: 13, weight: "bold" });
    b += cellGrid({
      matrix: Array.from({ length: 7 }, (_, i) => weights.slice(i * 7, i * 7 + 7)),
      x: 20,
      y: 260,
      size: 16,
      prefix: "dense-w",
      fill: (w) => (w >= 0 ? COLORS.cyan : COLORS.rose),
      alpha: (w) => (0.15 + 0.85 * Math.abs(w)).toFixed(2),
    });
    ["49 pesos propios", "+ 1 sesgo.", "Otra neurona:", "otros 49 pesos."].forEach((s, i) => {
      b += text(146, 276 + i * 19, s, { size: 12.5, fill: COLORS.muted });
    });
    b += text(20, 394, `Por neurona: 49 + 1 = ${dense.perNeuron}`, { size: 13 });
    b += text(20, 416, `Capa completa: 25 × ${dense.perNeuron} = ${dense.total}`, {
      size: 14,
      weight: "bold",
      fill: COLORS.rose,
    });

    // Conv2D half: one 3x3 window, the same 9 weights at every position.
    b += text(450, 28, "Conv2D", { size: 17, weight: "bold", anchor: "middle" });
    b += text(450, 45, "7×7 → 5×5 · ventana 3×3 compartida", {
      size: 11.5,
      fill: COLORS.muted,
      anchor: "middle",
    });
    b += cellGrid({ matrix: input, x: 320, y: 58, size: 22, fill: pixelFill });
    b += frameRect(320 + c * 22, 58 + r * 22, 66, 66, COLORS.cyan);
    const ox = 500 + c * 18 + 9;
    const oy = 90 + r * 18 + 9;
    b += line(320 + c * 22 + 66, 58 + r * 22 + 33, ox, oy, COLORS.cyan, 2, 0.8);
    b += cellGrid({
      matrix: map,
      x: 500,
      y: 90,
      size: 18,
      prefix: "conv-out",
      labels: true,
      filled: (i, j) => i * 5 + j <= pos,
      fill: () => COLORS.cyan,
      alpha: (v) => (0.15 + 0.6 * (v / maxMap)).toFixed(2),
    });
    b += frameRect(500 + c * 18, 90 + r * 18, 18, 18, COLORS.ink);
    b += text(545, 204, "mapa 5×5", { size: 11.5, fill: COLORS.muted, anchor: "middle" });
    b += text(320, 230, shiftTag, { size: 11.5, fill: phase ? COLORS.rose : COLORS.muted });
    b += text(320, 252, "Los mismos 9 pesos", { size: 13, weight: "bold" });
    b += cellGrid({
      matrix: data.kernel,
      x: 320,
      y: 260,
      size: 30,
      prefix: "kernel",
      labels: true,
      fill: weightFill(COLORS.cyan),
      textOn: (w) => w !== 0,
    });
    ["9 pesos + 1 sesgo,", "compartidos en las", "25 posiciones."].forEach((s, i) => {
      b += text(424, 276 + i * 19, s, { size: 12.5, fill: COLORS.muted });
    });
    b += text(320, 394, `Por filtro: 9 + 1 = ${conv.perFilter}`, { size: 13 });
    b += text(320, 416, `Capa completa: 1 filtro = ${conv.total}`, {
      size: 14,
      weight: "bold",
      fill: COLORS.green,
    });

    return {
      svg: svg("Densa frente a Conv2D: pesos propios frente a pesos compartidos", b),
      label:
        (phase ? "Casa desplazada 1 píxel. " : "") +
        `Posición ${pos + 1}/25 · Densa: neurona ${pos + 1} con 49 pesos propios · ` +
        `Conv2D: los mismos 9 pesos → ${map[r][c]}.`,
    };
  }

  // ---------- Lesson 2: one kernel sliding over the house ----------

  function renderKernel(frame) {
    const kernel = KERNELS.vertical;
    const out = convolve2d(HOUSE, kernel);
    const r = Math.floor(frame / 3);
    const c = frame % 3;
    const patch = windowAt(HOUSE, r, c, 3);
    const { terms, sum } = dotTerms(patch, kernel);
    let b = "";

    b += text(96, 50, "Imagen 5×5", { size: 13, weight: "bold", anchor: "middle" });
    b += text(252, 50, "Parche 3×3", { size: 13, weight: "bold", anchor: "middle" });
    b += text(370, 50, "Kernel", { size: 13, weight: "bold", anchor: "middle" });
    b += text(514, 50, "Mapa 3×3", { size: 13, weight: "bold", anchor: "middle" });

    b += cellGrid({ matrix: HOUSE, x: 16, y: 64, size: 32, fill: pixelFill });
    b += frameRect(16 + c * 32, 64 + r * 32, 96, 96, COLORS.cyan);
    b +=
      `<path d="M ${16 + c * 32 + 96} ${64 + r * 32 + 48} C 190 ${64 + r * 32 + 48}, 190 130, 210 130"` +
      ` fill="none" stroke="${COLORS.cyan}" stroke-width="2" stroke-dasharray="5 4"/>`;
    b += cellGrid({
      matrix: patch,
      x: 210,
      y: 88,
      size: 28,
      prefix: "patch",
      labels: true,
      fill: pixelFill,
      textOn: (v) => v !== 0,
    });
    b += text(311, 138, "×", { size: 22, anchor: "middle", fill: COLORS.muted });
    b += cellGrid({
      matrix: kernel,
      x: 328,
      y: 88,
      size: 28,
      prefix: "kernel",
      labels: true,
      fill: weightFill(COLORS.cyan),
      textOn: (w) => w !== 0,
    });
    b += text(437, 138, "→", { size: 22, anchor: "middle", fill: COLORS.muted });
    b += cellGrid({
      matrix: out,
      x: 460,
      y: 76,
      size: 36,
      prefix: "out",
      labels: true,
      filled: (i, j) => i * 3 + j <= frame,
      fill: () => COLORS.green,
      alpha: (v) => (0.2 + 0.25 * v).toFixed(2),
    });
    b += frameRect(460 + c * 36, 76 + r * 36, 36, 36, COLORS.rose);

    const spans = terms
      .map((t) => {
        const color = t.w ? COLORS.cyan : COLORS.muted;
        return `<tspan fill="${color}" font-weight="${t.w ? "bold" : "normal"}">${t.x}·${t.w}</tspan>`;
      })
      .join(" + ");
    b +=
      `<text x="300" y="272" font-size="14" text-anchor="middle" fill="${COLORS.ink}">` +
      `${spans} <tspan fill="${COLORS.rose}" font-weight="bold">= ${sum}</tspan></text>`;
    b += text(300, 302, `salida[${r + 1}, ${c + 1}] = ${sum}`, { size: 15, anchor: "middle", weight: "bold" });
    b += text(300, 334, "Cada píxel del parche se multiplica por su peso y todo se suma:", {
      size: 12.5,
      anchor: "middle",
      fill: COLORS.muted,
    });
    b += text(300, 352, "un solo número por posición de la ventana.", {
      size: 12.5,
      anchor: "middle",
      fill: COLORS.muted,
    });
    b += text(300, 414, "Kernel vertical ilustrativo (no entrenado) · paso 1 · sin relleno · sesgo 0", {
      size: 12,
      anchor: "middle",
      fill: COLORS.muted,
    });

    return {
      svg: svg("Un kernel 3×3 recorre la casa y llena el mapa de salida", b),
      label: `Posición ${frame + 1}/9: parche × kernel, suma = ${sum} → mapa[${r + 1}, ${c + 1}].`,
    };
  }

  // ---------- Lesson 3: several filters over the same input ----------

  const FILTER_COLORS = [COLORS.cyan, COLORS.green, COLORS.rose];

  function renderFilters(frame) {
    const final = frame >= 27;
    const f = final ? 3 : Math.floor(frame / 9);
    const p = final ? 8 : frame % 9;
    const maps = FILTER_BANK.map((filter) => convolve2d(HOUSE, filter.kernel));
    let b = "";

    b += text(80, 150, "Misma entrada", { size: 13, weight: "bold", anchor: "middle" });
    b += cellGrid({ matrix: HOUSE, x: 20, y: 160, size: 24, fill: pixelFill });
    if (!final) {
      const wr = Math.floor(p / 3);
      const wc = p % 3;
      b += frameRect(20 + wc * 24, 160 + wr * 24, 72, 72, FILTER_COLORS[f]);
    }

    FILTER_BANK.forEach((filter, i) => {
      const y0 = 22 + i * 134;
      const active = i === f;
      b += line(140, 220, 200, y0 + 45, active ? FILTER_COLORS[i] : COLORS.line, active ? 2.5 : 1.5);
      b += cellGrid({
        matrix: filter.kernel,
        x: 200,
        y: y0 + 12,
        size: 22,
        prefix: `k${i}`,
        fill: weightFill(FILTER_COLORS[i]),
      });
      b += text(292, y0 + 52, "→", { size: 20, anchor: "middle", fill: COLORS.muted });
      b += cellGrid({
        matrix: maps[i],
        x: 320,
        y: y0,
        size: 30,
        prefix: `map${i}`,
        labels: true,
        filled: (r, c) => final || i < f || (active && r * 3 + c <= p),
        fill: () => FILTER_COLORS[i],
        alpha: (v) => (0.15 + 0.22 * v).toFixed(2),
      });
      b += text(428, y0 + 30, `Filtro ${i + 1}: ${filter.name}`, { size: 13, weight: "bold" });
      b += text(428, y0 + 50, filter.hint, { size: 12, fill: COLORS.muted });
    });

    if (final) {
      b += frameRect(314, 16, 102, 102, COLORS.rose, true);
      b += frameRect(314, 284, 102, 102, COLORS.rose, true);
      b += text(428, 360, "Mapa 3 = Mapa 1:", { size: 13, weight: "bold", fill: COLORS.rose });
      b += text(428, 378, "redundancia", { size: 13, weight: "bold", fill: COLORS.rose });
    }
    b += text(300, 410, "El entrenamiento ajusta los pesos según la tarea:", {
      size: 12,
      anchor: "middle",
      fill: COLORS.muted,
    });
    b += text(300, 428, "los filtros pueden salir distintos o repetidos. Esta copia es ilustrativa.", {
      size: 12,
      anchor: "middle",
      fill: COLORS.muted,
    });

    const label = final
      ? "Los tres filtros vieron la misma casa. El mapa 3 repite el mapa 1: redundancia ilustrativa."
      : `Filtro ${f + 1} (${FILTER_BANK[f].name}), posición ${p + 1}/9: suma = ` +
        `${maps[f][Math.floor(p / 3)][p % 3]}.`;
    return { svg: svg("Tres filtros sobre la misma casa producen tres mapas", b), label };
  }

  const LESSONS = {
    comparison: { frameCount: 50, intervalMs: 800, title: "Densa frente a Conv2D", render: renderComparison },
    kernel: { frameCount: 9, intervalMs: 1400, title: "Un kernel recorre la imagen", render: renderKernel },
    filters: { frameCount: 28, intervalMs: 800, title: "Varios filtros, varios mapas", render: renderFilters },
  };

  // ---------- Player: frame state and timer ownership per widget ----------

  function createPlayer(o) {
    const isActive = o.isActive || (() => true);
    const state = { frame: 0, playing: !o.reducedMotion };
    let timer = null;

    const render = () => o.onFrame(state.frame, state.playing);
    const show = (f) => {
      state.frame = ((f % o.frameCount) + o.frameCount) % o.frameCount;
      render();
    };
    const stopTimer = () => {
      if (timer !== null) {
        o.scheduler.clearInterval(timer);
        timer = null;
      }
    };
    // A timer exists only while the user wants playback AND the slide is visible.
    const sync = () => {
      const run = state.playing && isActive();
      if (run && timer === null) timer = o.scheduler.setInterval(() => show(state.frame + 1), o.intervalMs || 900);
      if (!run) stopTimer();
    };
    const setPlaying = (playing) => {
      state.playing = playing;
      sync();
      render();
    };
    // Manual steps pause playback so the student can study the frame.
    const step = (f) => {
      state.playing = false;
      sync();
      show(f);
    };

    render();
    sync();
    return {
      state,
      sync,
      render,
      play: () => setPlaying(true),
      pause: () => setPlaying(false),
      toggle: () => setPlaying(!state.playing),
      next: () => step(state.frame + 1),
      prev: () => step(state.frame - 1),
      first: () => step(0),
      last: () => step(o.frameCount - 1),
      destroy() {
        state.playing = false;
        stopTimer();
      },
    };
  }

  // ---------- DOM mounting ----------

  const NAV_KEYS = new Set([" ", "Spacebar", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"]);
  const instances = new WeakMap();

  function mount(el, kind) {
    if (instances.has(el)) return instances.get(el);
    const key = kind || el.getAttribute("data-spatial-lesson");
    const lesson = LESSONS[key];
    if (!lesson) throw new Error(`Unknown spatial lesson: ${key}`);
    const doc = el.ownerDocument;
    const win = doc.defaultView;
    const make = (tag, cls) => {
      const node = doc.createElement(tag);
      node.classList.add(cls);
      return node;
    };

    el.classList.add("spatial-lesson");
    el.setAttribute("role", "group");
    el.setAttribute("aria-label", lesson.title);
    el.setAttribute("tabindex", "0");
    const stage = make("div", "spatial-stage");
    const label = make("p", "spatial-label");
    label.setAttribute("aria-live", "polite");
    const controls = make("div", "spatial-controls");
    const playButton = make("button", "spatial-button");
    const nextButton = make("button", "spatial-button");
    playButton.setAttribute("type", "button");
    nextButton.setAttribute("type", "button");
    nextButton.textContent = "Paso siguiente";
    controls.appendChild(playButton);
    controls.appendChild(nextButton);
    el.appendChild(stage);
    el.appendChild(label);
    el.appendChild(controls);

    const slide = el.closest ? el.closest(".slide") : null;
    const motion = win && win.matchMedia ? win.matchMedia("(prefers-reduced-motion: reduce)") : null;
    const isActive = () => (!slide || slide.classList.contains("is-active")) && !doc.hidden;

    const player = createPlayer({
      frameCount: lesson.frameCount,
      intervalMs: lesson.intervalMs,
      reducedMotion: Boolean(motion && motion.matches),
      isActive,
      scheduler: {
        setInterval: (fn, ms) => win.setInterval(fn, ms),
        clearInterval: (t) => win.clearInterval(t),
      },
      onFrame(frame, playing) {
        const out = lesson.render(frame);
        stage.innerHTML = out.svg;
        label.textContent = out.label;
        playButton.textContent = playing ? "Pausar" : "Reproducir";
      },
    });

    // Mouse clicks drop focus so deck arrow keys keep working; keyboard focus stays.
    const clickWith = (button, action) =>
      button.addEventListener("click", (event) => {
        action();
        if (event && event.detail > 0 && button.blur) button.blur();
      });
    clickWith(playButton, player.toggle);
    clickWith(nextButton, player.next);

    // Deck navigation keys never leave the widget while it has focus.
    el.addEventListener("keydown", (event) => {
      if (!NAV_KEYS.has(event.key)) return;
      event.stopPropagation();
      if (event.key === " " || event.key === "Spacebar") {
        const onButton = event.target && event.target.tagName === "BUTTON";
        if (!onButton) {
          event.preventDefault();
          player.toggle();
        }
        return;
      }
      event.preventDefault();
      if (event.key === "Home") player.first();
      else if (event.key === "End") player.last();
      else if (["ArrowRight", "ArrowDown", "PageDown"].includes(event.key)) player.next();
      else player.prev();
    });

    let observer = null;
    if (slide && win && win.MutationObserver) {
      observer = new win.MutationObserver(() => player.sync());
      observer.observe(slide, { attributes: true, attributeFilter: ["class"] });
    }
    if (motion && motion.addEventListener) {
      motion.addEventListener("change", (event) => {
        if (event.matches) player.pause();
      });
    }
    const onVisibility = () => player.sync();
    if (doc.addEventListener) doc.addEventListener("visibilitychange", onVisibility);

    const instance = {
      root: el,
      stage,
      label,
      playButton,
      nextButton,
      player,
      destroy() {
        player.destroy();
        if (observer) observer.disconnect();
        if (doc.removeEventListener) doc.removeEventListener("visibilitychange", onVisibility);
      },
    };
    instances.set(el, instance);
    return instance;
  }

  function autoMount(doc) {
    return Array.from(doc.querySelectorAll("[data-spatial-lesson]"))
      .filter((el) => LESSONS[el.getAttribute("data-spatial-lesson")])
      .map((el) => mount(el, el.getAttribute("data-spatial-lesson")));
  }

  return {
    HOUSE,
    KERNELS,
    FILTER_BANK,
    COLORS,
    LESSONS,
    padInto,
    windowAt,
    dotTerms,
    convolve2d,
    denseParams,
    convParams,
    denseWeights,
    comparisonData,
    createPlayer,
    mount,
    autoMount,
  };
});
